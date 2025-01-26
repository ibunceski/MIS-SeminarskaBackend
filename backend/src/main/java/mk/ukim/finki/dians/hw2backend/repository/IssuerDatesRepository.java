package mk.ukim.finki.dians.hw2backend.repository;

import mk.ukim.finki.dians.hw2backend.model.IssuerDates;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface IssuerDatesRepository extends JpaRepository<IssuerDates, String> {
    Optional<IssuerDates> findByIssuer(String issuer);
}
