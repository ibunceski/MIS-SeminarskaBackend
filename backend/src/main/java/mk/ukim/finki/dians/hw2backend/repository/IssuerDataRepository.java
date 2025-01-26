package mk.ukim.finki.dians.hw2backend.repository;

import mk.ukim.finki.dians.hw2backend.model.IssuerData;
import mk.ukim.finki.dians.hw2backend.model.IssuerDataKey;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface IssuerDataRepository extends JpaRepository<IssuerData, IssuerDataKey> {
    List<IssuerData> findByIssuer(String issuer);

    @Query("SELECT DISTINCT i.issuer FROM IssuerData i")
    List<String> getAllIssuers();
}
